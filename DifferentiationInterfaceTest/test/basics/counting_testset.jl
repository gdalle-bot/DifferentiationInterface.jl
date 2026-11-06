using Test

"""
    CountingTestSet

Test set which counts passes and failures instead of reporting them, to check that a testing function detects errors.

Nested test sets of the default type inherit this type and add their counts to their parent.
"""
mutable struct CountingTestSet <: Test.AbstractTestSet
    description::String
    passes::Int
    failures::Int
end

CountingTestSet(description::AbstractString; kwargs...) = CountingTestSet(description, 0, 0)

Test.record(ts::CountingTestSet, ::Test.Pass) = (ts.passes += 1; ts)
Test.record(ts::CountingTestSet, ::Union{Test.Fail, Test.Error}) = (ts.failures += 1; ts)
Test.record(ts::CountingTestSet, ::Test.Broken) = ts

function Test.record(ts::CountingTestSet, child::CountingTestSet)
    ts.passes += child.passes
    ts.failures += child.failures
    return ts
end

function Test.finish(ts::CountingTestSet)
    if Test.get_testset_depth() > 0
        parent = Test.get_testset()
        parent isa CountingTestSet && Test.record(parent, ts)
    end
    return ts
end

"""
    count_tests(f)

Run `f()` inside a [`CountingTestSet`](@ref) and return the numbers of passes and failures.
"""
function count_tests(f)
    ts = @testset CountingTestSet "counting" begin
        f()
    end
    return (; ts.passes, ts.failures)
end
