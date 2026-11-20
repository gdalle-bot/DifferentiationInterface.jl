function DIT.test_jet(
        backend::AbstractADType,
        ts::DIT.TripleScenario;
        subset::Symbol,
        ignored_modules,
        function_filter,
    )
    subset == :none && return nothing
    calls = DIT.prepared_calls(backend, ts)
    if subset == :full
        foreach((calls.preparation, calls.unprepared)) do (f, args)
            test_opt(f, Base.typesof(args...); ignored_modules, function_filter)
        end
    end
    if subset in (:prepared, :full)
        f, args = calls.prepared
        test_opt(f, Base.typesof(args...); ignored_modules, function_filter)
    end
    return nothing
end
