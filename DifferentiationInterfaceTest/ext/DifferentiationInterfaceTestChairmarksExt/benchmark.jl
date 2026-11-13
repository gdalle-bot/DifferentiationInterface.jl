@kwdef struct BenchmarkResult
    prepared::Benchmark = failed_bench()
    preparation::Benchmark = failed_bench()
    unprepared::Benchmark = failed_bench()
end

@kwdef struct CallsResult
    prepared::Int = -1
    preparation::Int = -1
    unprepared::Int = -1
end

function benchmark_triple(
        backend::AbstractADType, ts::DIT.TripleScenario; subset::Symbol, seconds::Real
    )
    calls = DIT.prepared_calls(backend, ts)
    f_prep, args_prep = calls.preparation
    f_unprep, args_unprep = calls.unprepared
    f, args = calls.prepared
    prepared = @be f(args...) seconds = seconds
    if subset == :full
        preparation = @be f_prep(args_prep...) seconds = seconds
        unprepared = @be f_unprep(args_unprep...) seconds = seconds
        return BenchmarkResult(; prepared, preparation, unprepared)
    else
        return BenchmarkResult(; prepared)
    end
end

function count_calls_triple(backend::AbstractADType, ts::DIT.TripleScenario)
    exec_s = DIT.exec_scenario(ts)
    cc = CallCounter(DIT.func(exec_s))
    with_counter(s) = DIT.Scenario(
        DIT.operator(s), cc, DIT.pre_backend_args(s), DIT.post_backend_args(s);
        result = DIT.result(s), name = DIT.name(s),
    )
    new_ts = DIT.TripleScenario(
        with_counter(exec_s);
        prep = with_counter(DIT.prep_scenario(ts)),
        flush = with_counter(DIT.flush_scenario(ts)),
    )
    calls = DIT.prepared_calls(backend, new_ts)
    # `prepared_calls` makes a deep copy of the counter
    new_cc = first(last(calls.preparation))
    reset_count!(new_cc)
    f_prep, args_prep = calls.preparation
    f_prep(args_prep...)
    preparation = reset_count!(new_cc)
    f, args = calls.prepared
    f(args...)
    prepared = reset_count!(new_cc)
    f_unprep, args_unprep = calls.unprepared
    f_unprep(args_unprep...)
    unprepared = reset_count!(new_cc)
    return CallsResult(; prepared, preparation, unprepared)
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
    @assert subset in (:full, :prepared)
    op = DIT.operator(ts)

    bench_success = true
    bench_result = try
        benchmark_triple(backend, ts; subset, seconds = benchmark_seconds)
    catch exception
        bench_success = false
        logging && @warn "Error during benchmarking" backend ts exception
        BenchmarkResult()
    end
    benchmark_test && @test bench_success

    if count_calls
        count_success = true
        calls_result = try
            count_calls_triple(backend, ts)
        catch exception
            count_success = false
            logging && @warn "Error during call counting" backend ts exception
            CallsResult()
        end
        benchmark_test && @test count_success
    else
        calls_result = CallsResult()
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
