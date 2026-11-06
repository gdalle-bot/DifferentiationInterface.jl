module DifferentiationInterfaceTestChairmarksExt

using ADTypes: AbstractADType
using Chairmarks: @be, Benchmark, Sample
import DifferentiationInterfaceTest as DIT
using DifferentiationInterfaceTest:
    CallCounter, DifferentiationBenchmarkDataRow, DifferentiationBenchmark, reset_count!
using Test: Test, @test

function failed_bench()
    evals = 0.0
    time = NaN
    allocs = NaN
    bytes = NaN
    gc_fraction = NaN
    compile_fraction = NaN
    recompile_fraction = NaN
    warmup = NaN
    checksum = NaN
    sample = Sample(
        evals,
        time,
        allocs,
        bytes,
        gc_fraction,
        compile_fraction,
        recompile_fraction,
        warmup,
        checksum,
    )
    return Benchmark([sample])
end

function record!(
        data::DifferentiationBenchmark;
        backend::AbstractADType,
        scenario::DIT.TripleScenario,
        operator::String,
        prepared::Union{Nothing, Bool},
        bench::Benchmark,
        calls::Integer,
        aggregation,
    )
    row = DifferentiationBenchmarkDataRow(;
        backend = backend,
        scenario = scenario,
        operator = Symbol(operator),
        prepared = prepared,
        calls = calls,
        samples = length(bench.samples),
        evals = Int(bench.samples[1].evals),
        time = aggregation(getfield.(bench.samples, :time)),
        allocs = aggregation(getfield.(bench.samples, :allocs)),
        bytes = aggregation(getfield.(bench.samples, :bytes)),
        gc_fraction = aggregation(getfield.(bench.samples, :gc_fraction)),
        compile_fraction = aggregation(getfield.(bench.samples, :compile_fraction)),
    )
    return push!(data.rows, row)
end

include("benchmark.jl")

end
