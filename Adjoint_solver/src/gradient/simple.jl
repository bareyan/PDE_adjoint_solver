## Finite differences: one extra state solve per parameter (forward difference).
function FD(p::Problem, loss, V, target; eps = 1e-8)
    grad = zeros(Float64, p.N)
    base = loss(V, target)
    for i in 1:p.N
        V_i = copy(V); V_i[i] += eps
        grad[i] = (loss(V_i, target) - base) / eps
    end
    return grad
end

## Forward-mode AD: dual numbers, one tangent direction per parameter.
forwarddiff_gradient(loss, V) = ForwardDiff.gradient(loss, V)

## Reverse-mode AD (Zygote): one backward pass, rules consumed natively.
zygote_gradient(loss, V) = first(Zygote.gradient(loss, V))

## Reverse-mode AD (Mooncake): one backward pass; grads = (∂loss/∂f, ∂loss/∂V).
function mooncake_gradient(loss, V)
    cache = Mooncake.prepare_gradient_cache(loss, V)
    _, grads = Mooncake.value_and_gradient!!(cache, loss, V)
    return grads[2]
end
