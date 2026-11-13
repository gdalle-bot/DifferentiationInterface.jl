"""
    supports_inplace_output(x)

Return `true` if a buffer similar to `x` can be overwritten with `x`, by actually trying it.
"""
supports_inplace_output(x::Tuple) = all(supports_inplace_output, x)
function supports_inplace_output(x)
    try
        copyto!(mysimilar(x), x)
        return true
    catch
        return false
    end
end

"""
    make_buffers(op, r::Result)

Return the tuple of output buffers taken by `op`, built from the reference result `r`.
"""
function make_buffers(op, r::Result)
    is_inplace(op) || return ()
    return map(k -> mysimilar(getorder(r, k)), filter(!iszero, returned_orders(op)))
end

"""
    change_operator(s::Scenario, new_op)

Return a scenario identical to `s` but with operator `new_op`, which must be in the same family.

The output buffers are rebuilt from the reference result of `s`, and the primal output `y` (for in-place functions) is copied.
"""
function change_operator(s::Scenario, new_op)
    if standard(new_op) !== standard(operator(s))
        throw(ArgumentError("`$new_op` is not in the same family as `$(operator(s))`"))
    end
    r = result(s)
    if isnothing(r)
        throw(ArgumentError("Changing the operator requires a reference result"))
    end
    check_result(new_op, r)
    return Scenario(
        new_op,
        func(s),
        (map(deepcopy, value_args(s))..., make_buffers(new_op, r)...),
        post_backend_args(s);
        result = r,
        name = name(s),
    )
end

with_value(s::Scenario) = change_operator(s, with_value(operator(s)))
without_value(s::Scenario) = change_operator(s, without_value(operator(s)))
inplace(s::Scenario) = change_operator(s, inplace(operator(s)))
outofplace(s::Scenario) = change_operator(s, outofplace(operator(s)))

"""
    can_be_inplace(s::Scenario)

Return `true` if the outputs of the operator in `s` can be written to buffers.
"""
function can_be_inplace(s::Scenario)
    r = result(s)
    isnothing(r) && return false
    orders = filter(!iszero, returned_orders(with_value(operator(s))))
    return all(k -> supports_inplace_output(getorder(r, k)), orders)
end

"""
    all_variants(s::Scenario)

Return the vector of scenarios with every operator in the family of `s`: with and without value, out-of-place and in-place (when the outputs support it).

The reference result of `s` must contain every differentiation order.
"""
function all_variants(s::Scenario)
    std = standard(operator(s))
    ops = can_be_inplace(s) ? (std, inplace(std), with_value(std), inplace(with_value(std))) :
        (std, with_value(std))
    return [change_operator(s, op) for op in ops]
end
