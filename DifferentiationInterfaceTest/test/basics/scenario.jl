module ScenarioTests

using DifferentiationInterface
import DifferentiationInterfaceTest as DIT
using DifferentiationInterfaceTest: Scenario, Result, @scenario
using Test

f_scalar(x, c = 1.0) = c * sum(abs2, x)
f_vec!(y, x) = (y .= 2 .* x; nothing)

function macro_error(ex)
    try
        macroexpand(@__MODULE__, ex)
        return nothing
    catch e
        return e isa LoadError ? e.error : e
    end
end

same_fields(s1::Scenario, s2::Scenario) = all(
    getfield(s1, k) === getfield(s2, k) for k in fieldnames(Scenario)
)

@testset "Macro" begin
    x, g, c = [1.0, 2.0], [2.0, 4.0], Constant(1.0)
    r = Result(5.0, g, nothing)

    s1 = @scenario "named" value_and_gradient(f_scalar, missing, x, c) r
    s2 = Scenario(value_and_gradient, f_scalar, (), (x, c); result = r, name = "named")
    @test same_fields(s1, s2)

    s1 = @scenario value_and_gradient!(f_scalar, g, missing, x) r
    s2 = Scenario(value_and_gradient!, f_scalar, (g,), (x,); result = r)
    @test same_fields(s1, s2)
    @test isnothing(DIT.name(s1))

    s1 = @scenario gradient(f_scalar, missing, x)
    @test isnothing(DIT.result(s1))

    # splatting
    post = (x, c)
    s1 = @scenario gradient(f_scalar, missing, post...) r
    @test DIT.post_backend_args(s1) === post

    # keyword constructor
    s2 = Scenario(; operator = gradient, func = f_scalar, post_backend_args = post, result = r)
    @test same_fields(s1, s2)

    @test macro_error(:(@scenario gradient(f_scalar, x))) isa ArgumentError
    @test macro_error(:(@scenario gradient(f_scalar, missing, missing))) isa ArgumentError
    @test macro_error(:(@scenario gradient(f_scalar, missing, x; strict = Val(true)))) isa
        ArgumentError
    @test macro_error(:(@scenario "a" "b" gradient(f_scalar, missing, x))) isa ArgumentError
    @test macro_error(:(@scenario gradient(f_scalar, missing, x) r r)) isa ArgumentError
end

@testset "Validation" begin
    x, y, dx = [1.0, 2.0], [0.0, 0.0], ([1.0, 0.0],)
    @test_throws ArgumentError("`sum` is not a supported DifferentiationInterface operator") Scenario(
        sum, f_scalar, (), (x,)
    )
    # wrong number of buffers
    @test_throws ArgumentError(
        "Operator `gradient!` takes 1 output buffer(s) before the backend, plus `y` for in-place functions, but got 0 argument(s)",
    ) Scenario(gradient!, f_scalar, (), (x,))
    @test_throws ArgumentError(
        "Operator `value_gradient_and_hessian!` takes 2 output buffer(s) before the backend, plus `y` for in-place functions, but got 1 argument(s)",
    ) Scenario(value_gradient_and_hessian!, f_scalar, (x,), (x,))
    # in-place functions only for some operators
    @test Scenario(jacobian, f_vec!, (y,), (x,)) isa Scenario
    @test_throws ArgumentError("Operator `gradient` does not support in-place functions") Scenario(
        gradient, f_vec!, (y,), (x,)
    )
    # tangents
    @test Scenario(pushforward, f_vec!, (y,), (x, dx)) isa Scenario
    @test_throws ArgumentError(
        "Operator `pushforward` needs at least 2 argument(s) after the backend, got 1"
    ) Scenario(pushforward, f_vec!, (y,), (x,))
    @test_throws ArgumentError("Operator `pushforward` needs a tuple of tangents after the input `x`") Scenario(
        pushforward, f_vec!, (y,), (x, x)
    )
    # contexts
    @test_throws ArgumentError("Extra arguments must be contexts, got Float64") Scenario(
        gradient, f_scalar, (), (x, 1.0)
    )
    # result
    @test_throws ArgumentError(
        "Operator `value_and_gradient` returns order 0 but the reference `Result` does not contain it",
    ) Scenario(
        value_and_gradient, f_scalar, (), (x,); result = Result(; order1 = x)
    )
end

@testset "Accessors" begin
    x, y, dx, dy = [1.0, 2.0], [0.0, 0.0], ([1.0, 0.0],), ([0.0, 1.0],)
    c, k = Constant(1.0), Cache([0.0])

    s = @scenario value_and_pullback!(f_vec!, y, dy, missing, x, dy, c, k) Result(y, dy, nothing)
    @test DIT.has_inplace_function(s)
    @test DIT.value_args(s) === (y,)
    @test DIT.output_buffers(s) === (dy,)
    @test DIT.primal_inputs(s) === (x,)
    @test DIT.primal_input(s) === x
    @test DIT.tangents(s) === dy
    @test DIT.contexts(s) === (c, k)
    @test DIT.primal_output(s) === y
    @test DIT.protected_args(s) === (x, dy, c)

    s = @scenario gradient(f_scalar, missing, x, c) Result(5.0, x, nothing)
    @test !DIT.has_inplace_function(s)
    @test DIT.value_args(s) === ()
    @test DIT.output_buffers(s) === ()
    @test DIT.tangents(s) === nothing
    @test DIT.contexts(s) === (c,)
    @test DIT.primal_output(s) === 5.0

    s = @scenario gradient(f_scalar, missing, x)
    @test DIT.primal_output(s) === nothing
end

@testset "Display" begin
    s = @scenario "my scenario" gradient(f_scalar, missing, [1.0])
    @test string(s) == "my scenario"
    s = @scenario gradient!(f_scalar, [1.0], missing, [1.0], Constant(1.0))
    @test startswith(string(s), "gradient!(f_scalar, ::Vector{Float64}, backend, ::Vector{Float64}, ::")
end

end
