using Enzyme

## Finite Differences
function FD(p::Problem, loss, V, target;eps=1e-8)
    grad = zeros(Float64, p.N)
    for i in 1:p.N
        V_i = copy(V); V_i[i] += eps
        grad[i] = (loss(V_i, target) - loss(V, target)) / eps
    end
    return grad
end

## Automatic Differentiation
function Forward_AD(p::Problem, loss, V, target)
    res = zeros(p.N)
    for i in 1:p.N
        v = zeros(p.N)
        v[i] = 1
        (col,) = autodiff(set_runtime_activity(Forward), loss, Duplicated(V, v), Const(target))
        res[i] = col
    end
    # gradient(Forward, loss, V, Const(target))
    return res
end

function Backward_AD(p::Problem, loss, V, target)
    v = zeros(p.N)
    autodiff(set_runtime_activity(Reverse), loss, Active,  Duplicated(V, v), Const(target))
    return v
end