"""
    make_buffers(op, r::Result)

Return the tuple of output buffers taken by `op`, built from the reference result `r`.
"""
function make_buffers(@nospecialize(op), @nospecialize(r::Result))
    is_inplace(op) || return ()
    return map(k -> mysimilar(getorder(r, k)), filter(!iszero, returned_orders(op)))
end

"""
    change_operator(s::Scenario, new_op)

Return a scenario identical to `s` but with operator `new_op`, which must be in the same family.

The output buffers are rebuilt from the reference result of `s`, and the primal output `y` (for in-place functions) is copied.
"""
function change_operator(@nospecialize(s::Scenario), @nospecialize(new_op))
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

with_value(@nospecialize(s::Scenario)) = change_operator(s, with_value(operator(s)))
without_value(@nospecialize(s::Scenario)) = change_operator(s, without_value(operator(s)))
inplace(@nospecialize(s::Scenario)) = change_operator(s, inplace(operator(s)))
outofplace(@nospecialize(s::Scenario)) = change_operator(s, outofplace(operator(s)))

"""
    all_variants(s::Scenario; include_inplace::Bool)

Return the vector of scenarios with every operator in the family of `s`: with and without value, out-of-place and, if `include_inplace` is `true`, in-place.

The reference result of `s` must contain every differentiation order.
In-place variants require outputs which can be overwritten, such as mutable arrays: this is up to the caller.
"""
function all_variants(@nospecialize(s::Scenario); include_inplace::Bool)
    std = standard(operator(s))
    ops = if include_inplace
        (std, inplace(std), with_value(std), inplace(with_value(std)))
    else
        (std, with_value(std))
    end
    return [change_operator(s, op) for op in ops]
end

"""
    place_variants(s::Scenario; inplace_op::Bool)

Return the vector of scenarios with the operators of the family of `s` which are in-place (if `inplace_op` is `true`) or out-of-place (otherwise), with and without value.
"""
function place_variants(@nospecialize(s::Scenario); inplace_op::Bool)
    std = standard(operator(s))
    ops = inplace_op ? (inplace(std), inplace(with_value(std))) : (std, with_value(std))
    return [change_operator(s, op) for op in ops]
end
