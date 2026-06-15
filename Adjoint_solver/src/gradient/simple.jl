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


## AD methods (forward/reverse) using Enzyme
function Forward_AD(loss, V)
    res = zeros(length(V))
    for i in 1:length(V)
        v = zeros(length(V))
        v[i] = 1
        (col,) = autodiff(set_runtime_activity(Forward), Const(loss), Duplicated(V, v))
        res[i] = col
    end
    return res
    # grad, = gradient(set_runtime_activity(Forward), loss, Duplicated(V); chunk=Val(8))
    # return grad
end

function Backward_AD(loss, V)
    v = zeros(length(V))
    autodiff(set_runtime_activity(Reverse), Const(loss), Active,  Duplicated(V, v))
    return v
end
