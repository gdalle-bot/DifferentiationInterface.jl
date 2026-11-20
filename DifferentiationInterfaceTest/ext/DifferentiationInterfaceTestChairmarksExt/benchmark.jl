@kwdef struct TripleBenchmarkResult
    prepared::Benchmark = failed_bench()
    preparation::Benchmark = failed_bench()
    unprepared::Benchmark = failed_bench()
end

@kwdef struct TripleCallsResult
    prepared::Int = -1
    preparation::Int = -1
    unprepared::Int = -1
end

function benchmark_triple(
        backend::AbstractADType, ts::DIT.TripleScenario; subset::Symbol, seconds::Real
    )
    @assert subset in (:prepared, :full)
    calls = DIT.prepared_calls(backend, ts)
    f_prep, args_prep = calls.preparation
    f_unprep, args_unprep = calls.unprepared
    f, args = calls.prepared
    prepared = @be f(args...) seconds = seconds
    if subset == :full
        preparation = @be f_prep(args_prep...) seconds = seconds
        unprepared = @be f_unprep(args_unprep...) seconds = seconds
        return TripleBenchmarkResult(; prepared, preparation, unprepared)
    else
        return TripleBenchmarkResult(; prepared)
    end
end

"""
    call_count((f, args))

Call `f(args...)` and return the number of times that the `CallCounter` in first position of `args` was called.
"""
function call_count((f, args))
    cc = first(args)
    reset_count!(cc)
    f(args...)
    return reset_count!(cc)
end

function count_calls_triple(backend::AbstractADType, ts::DIT.TripleScenario)
    # each distinct function gets its own counter, and identical functions share one
    counters = IdDict{Any, CallCounter}()
    # TODO: use the new version of `change_function` here
    function with_counter(s)
        cc = get!(() -> CallCounter(DIT.func(s)), counters, DIT.func(s))
        return DIT.Scenario(
            DIT.operator(s), cc, DIT.pre_backend_args(s), DIT.post_backend_args(s);
            result = DIT.result(s), name = DIT.name(s),
        )
    end
    new_ts = DIT.TripleScenario(
        with_counter(DIT.exec_scenario(ts));
        prep = with_counter(DIT.prep_scenario(ts)),
        flush = with_counter(DIT.flush_scenario(ts)),
    )
    # `prepared_calls` makes a deep copy, which preserves the sharing of counters
    calls = DIT.prepared_calls(backend, new_ts)
    return TripleCallsResult(;
        preparation = call_count(calls.preparation),
        prepared = call_count(calls.prepared),
        unprepared = call_count(calls.unprepared),
    )
end

function DIT.run_benchmark!(
        data::DifferentiationBenchmark,
        backend::AbstractADType,
        ts::DIT.TripleScenario;
        logging::Bool,
        subset::Symbol,
        count_calls::Bool,
        benchmark_test::Bool,
        benchmark_seconds::Real,
        benchmark_aggregation,
    )
    subset == :none && return nothing
    op = DIT.operator(ts)

    bench_success = true
    bench_result = try
        benchmark_triple(backend, ts; subset, seconds = benchmark_seconds)
    catch exception
        bench_success = false
        logging && @warn "Error during benchmarking" backend ts exception
        TripleBenchmarkResult()
    end
    benchmark_test && @test bench_success

    if count_calls
        count_success = true
        calls_result = try
            count_calls_triple(backend, ts)
        catch exception
            count_success = false
            logging && @warn "Error during call counting" backend ts exception
            TripleCallsResult()
        end
        benchmark_test && @test count_success
    else
        calls_result = TripleCallsResult()
    end

    record!(
        data;
        backend,
        scenario = ts,
        operator = string(nameof(op)),
        prepared = true,
        bench = bench_result.prepared,
        calls = calls_result.prepared,
        aggregation = benchmark_aggregation,
    )
    if subset == :full
        record!(
            data;
            backend,
            scenario = ts,
            operator = string(nameof(DIT.preparator(op))),
            prepared = nothing,
            bench = bench_result.preparation,
            calls = calls_result.preparation,
            aggregation = benchmark_aggregation,
        )
        record!(
            data;
            backend,
            scenario = ts,
            operator = string(nameof(op)),
            prepared = false,
            bench = bench_result.unprepared,
            calls = calls_result.unprepared,
            aggregation = benchmark_aggregation,
        )
    end
    return nothing
end
